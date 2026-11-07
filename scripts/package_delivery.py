#!/usr/bin/env python3
"""Package immutable inputs, portable models, outputs and code without credentials/envs."""
import tarfile
from movens_sim.util import ROOT,read,write,sha,now

manifest=read(ROOT/'configs/manifest.json');paths=[]
for name in ['README.md','AGENTS.md','agent.md','pyproject.toml','requirements.lock.txt','.gitignore','src','configs','prompts','scripts','tests','docs','runs/batch_v1']:
    p=ROOT/name
    paths.extend([p] if p.is_file() else [q for q in p.rglob('*') if q.is_file()])
for e in manifest['entries']:paths.extend(p for p in (ROOT/'data'/f"episode_{e['episode']:06}").rglob('*') if p.is_file())
paths.extend(p for p in (ROOT/'data').glob('*') if p.is_file())
paths=sorted(set(p for p in paths if '__pycache__' not in p.parts and not any(part.endswith('.egg-info') for part in p.parts) and p.suffix not in ['.env','.pyc'] and not p.name.endswith('.resume_backup')))
record=dict(created_utc=now(),credentials_included=False,environment_included=False,relative_asset_closures=True,files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in paths])
write(ROOT/'dist/package_manifest.json',record);target=ROOT/'dist/movens_sim_batch_v1.tar.gz';target.parent.mkdir(exist_ok=True)
with tarfile.open(target,'w:gz',compresslevel=6) as archive:
    for path in paths:archive.add(path,arcname='Movens-Sim/'+str(path.relative_to(ROOT)),recursive=False)
    archive.add(ROOT/'dist/package_manifest.json',arcname='Movens-Sim/package_manifest.json')
target.with_suffix(target.suffix+'.sha256').write_text(sha(target)+'  '+target.name+'\n')
print(str(target),target.stat().st_size,'bytes',flush=True)
