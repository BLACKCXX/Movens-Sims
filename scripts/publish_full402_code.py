#!/usr/bin/env python3
"""Publish tested, hash-bound compatible code; existing workers stay untouched."""
from pathlib import Path
import argparse
import re
import sys
import shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from movens_sim.util import ROOT,read,write,sha,digest,now
from movens_sim.reconstruction.workflow import identity

p=argparse.ArgumentParser();p.add_argument('--version',required=True);p.add_argument('--validation',required=True);a=p.parse_args()
if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',a.version):raise ValueError('Invalid version')
root=ROOT/'runs/full402_h0_v1';version=root/'code_versions'/a.version;c=read(root/'resolved_config.json')
source=next(p for p in sorted(root.glob('episode_*')) if (p/'run_identity.json').exists());new=identity(source,c);validation=read(root/a.validation)
if not validation.get('passed') or validation['code_sha256']!=digest(new['code']):raise ValueError('This exact code has no passing validation receipt')
if version.exists():raise RuntimeError('Published snapshots are immutable')
for folder in ('src','prompts','configs'):shutil.copytree(ROOT/folder,version/folder,ignore=shutil.ignore_patterns('__pycache__'))
(version/'scripts').mkdir();shutil.copy2(ROOT/'scripts/movens-sim',version/'scripts/movens-sim');shutil.copy2(ROOT/'pyproject.toml',version/'pyproject.toml')
hashes={str(p.relative_to(version)):sha(p) for p in version.rglob('*') if p.is_file()};manifest=f'code_versions/{a.version}_hashes.json';write(root/manifest,hashes)
previous={digest(read(p/'run_identity.json')['code']) for p in root.glob('episode_*') if (p/'run_identity.json').exists()}
oldtransition=read(root/'rolling_code_transition.json') if (root/'rolling_code_transition.json').exists() else None
if oldtransition:
    previous.update(oldtransition['from_code_sha256s']);write(root/f'code_versions/previous_transition_{sha(root/"rolling_code_transition.json")[:12]}.json',oldtransition)
write(root/'rolling_code_transition.json',dict(from_code_sha256s=sorted(previous),to_code_sha256=digest(new['code']),reason='Supported nonintersecting geometric search seeds for bad FK priors; distinguish constraints from system failures; completed-delivery integrity check',published_utc=now(),validation=a.validation,budgets_reset=False,existing_workers_unchanged=True))
write(root/'active_code.json',dict(code_root='code_versions/'+a.version,hash_manifest=manifest,reason='Tested rolling compatible continuation; earlier snapshots and completed scene bindings retained'))
print('Published '+a.version+'; no running worker or budget modified')
