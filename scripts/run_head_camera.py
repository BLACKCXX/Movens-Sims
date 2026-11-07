"""Canonical pilot launcher with recursive receipt checking and optional probes."""
from __future__ import annotations

import argparse
import fcntl
import shutil
import sys
from pathlib import Path

from movens_sim import head_camera
from movens_sim.pipeline import receipt_valid
from movens_sim.util import ROOT,append,code_identity,digest,now,read,sha,write


def repair_nested_receipts(run,identity):
    """Never skip a missing case artifact just because its receipt file survives."""
    run=Path(run)
    if not (run/'candidate_index.json').exists():return []
    broken=[]
    for row in read(run/'candidate_index.json'):
        folder=Path(row['folder'])
        if not receipt_valid(folder,'fit',identity):broken.append(str(folder))
    if not broken:return []
    archive=run/'repair_history'/now().replace(':','').replace('+','_');archive.mkdir(parents=True)
    for stage in ('fit','validate','render'):
        source=run/'receipts'/(stage+'.json')
        if source.exists():shutil.move(source,archive/source.name)
    if (run/'result.json').exists():
        shutil.copy2(run/'result.json',archive/'previous_result.json');result=read(run/'result.json');result['processing_complete']=False;write(run/'result.json',result)
    write(archive/'reason.json',dict(broken_case_receipts=broken,action='Invalidate dependent receipts, preserve their previous versions; existing candidate checkpoints retain consumed evaluation budgets. Full run will regenerate missing outputs.',utc=now()))
    return broken


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--config',default='configs/head_camera_pilot_v1.json');parser.add_argument('--episode',type=int);parser.add_argument('--run-root');parser.add_argument('--resume',action='store_true');parser.add_argument('--dry-run',action='store_true');parser.add_argument('--stage');parser.add_argument('--with-mount-refinement',action='store_true')
    args,_=parser.parse_known_args(argv)
    if '--help' in argv or '-h' in argv:
        print('Additional launcher flag: --with-mount-refinement (versioned bounded camera-sensitivity audit after a complete pilot)')
        head_camera.main([x for x in argv if x!='--with-mount-refinement']);return
    config=read(ROOT/args.config)
    if args.run_root:config['run_root']=args.run_root
    if args.episode is not None and not args.dry_run:
        run=ROOT/config['run_root']/f'episode_{args.episode:06}'
        if (run/'run_identity.json').exists() and not args.resume:raise ValueError('Existing experiment: use --resume or a new --run-root; histories and consumed budgets will not be overwritten')
        if args.resume and (run/'run_identity.json').exists():
            identity=read(run/'run_identity.json')
            if identity['code_hashes']!=code_identity() or identity['config_sha256']!=digest(config):raise ValueError('Input/code/config mismatch; no repair attempted')
            lock=(run/'.pilot.lock').open('a')
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);broken=repair_nested_receipts(run,identity)
                if broken:print('Restoring incomplete case outputs from checkpoints: '+str(broken),flush=True)
            finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        run.mkdir(parents=True,exist_ok=True)
        append(run/'launches.jsonl',dict(utc=now(),argv=argv,launcher_sha256=sha(__file__),shell_launcher_sha256=sha(ROOT/'scripts/head-camera-pilot'),refinement_scheduler_sha256=sha(ROOT/'scripts/head_camera_refine.py') if args.with_mount_refinement else None))
    head_camera.main([x for x in argv if x!='--with-mount-refinement'])
    if args.with_mount_refinement and not args.dry_run and args.stage in (None,'render'):
        from head_camera_refine import run as refine
        summary=refine(config,args.episode,args.resume)
        print(dict(refinement_static_passed=summary['static_passed'],total_numeric_evaluations=summary['total_numeric_evaluations']),flush=True)


if __name__=='__main__':main()
