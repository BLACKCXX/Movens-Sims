#!/usr/bin/env python3
"""Publish a compatible snapshot for future workers without editing active ones."""
from pathlib import Path
import sys
import shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from movens_sim.util import ROOT,read,write,sha,digest,now
from movens_sim.reconstruction.workflow import identity

root=ROOT/'runs/full402_h0_v1';version=root/'code_versions/constraint_baseline_v3'
if version.exists():raise RuntimeError('Published snapshots are immutable')
for folder in ('src','prompts','configs'):shutil.copytree(ROOT/folder,version/folder,ignore=shutil.ignore_patterns('__pycache__'))
(version/'scripts').mkdir();shutil.copy2(ROOT/'scripts/movens-sim',version/'scripts/movens-sim');shutil.copy2(ROOT/'pyproject.toml',version/'pyproject.toml')
hashes={str(p.relative_to(version)):sha(p) for p in version.rglob('*') if p.is_file()};write(root/'code_versions/constraint_baseline_v3_hashes.json',hashes)
c=read(root/'resolved_config.json');source=next(p for p in sorted(root.glob('episode_*')) if (p/'run_identity.json').exists());new=identity(source,c)
previous={digest(read(p/'run_identity.json')['code']) for p in root.glob('episode_*') if (p/'run_identity.json').exists()}
write(root/'rolling_code_transition.json',dict(from_code_sha256s=sorted(previous),to_code_sha256=digest(new['code']),reason='Report invalid FK baselines without accepting physical violations; constraint failures do not trigger a system-wide circuit; validate completed delivery integrity',published_utc=now(),budgets_reset=False,existing_workers_unchanged=True))
write(root/'active_code.json',dict(code_root='code_versions/constraint_baseline_v3',hash_manifest='code_versions/constraint_baseline_v3_hashes.json',reason='Rolling checked code continuation; v1/v2 snapshots and completed scene bindings retained'))
print('Published constraint_baseline_v3 for future workers; existing workers continue their exact snapshot')
