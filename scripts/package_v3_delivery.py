#!/usr/bin/env python3
"""Portable frozen run, source, counters and evidence; excludes render-search cache."""
import argparse
import json
from pathlib import Path
import sqlite3
import tempfile
import zipfile

from movens_sim.util import ROOT,read,write,sha,now


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();run=Path(a.run).resolve();out=Path(a.output).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    if out.is_relative_to(run):raise ValueError('Archive must be outside the run')
    files=[]
    for folder in ('src','scripts','configs','prompts','tests'):
        files.extend(path for path in (ROOT/folder).rglob('*') if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc')
    for name in ('AGENTS.md','agent.md','pyproject.toml','requirements-v3.lock.txt','docs/mount_joint_v3.md'):files.append(ROOT/name)
    files.extend(path for path in run.rglob('*') if path.is_file() and 'evaluation_cache' not in path.parts and path.name!='package_verification.json' and not path.name.startswith('.') and not path.name.endswith('.tmp'))
    config=read(run/'resolved_config.json');db=ROOT/'runs/_task_budgets'/(config['task_group_id']+'.sqlite');registry=ROOT/'runs/_task_budgets'/f"mount_joint_v3_episode_{read(run/'source_manifest.json')['episode']:06}.json";files.append(registry)
    manifest={str(path.relative_to(ROOT)):dict(sha256=sha(path),bytes=path.stat().st_size) for path in sorted(set(files))}
    relative=str(run.relative_to(ROOT));readme=f'''# Movens-Sim frozen episode delivery\n\nThis is a static 3D simulation candidate, not real capture, interaction replay, calibrated hardware or validated real dynamics. Quality did NOT pass; appearance was blocked. See `{relative}/report.md` and `index.html`.\n\nRequires Python 3.12, system FFmpeg/ffprobe and a working EGL/OpenGL driver. Install requirements-v3.lock.txt and the local package. No source Movens-Interact directory or API key is needed to load/render the frozen scene:\n\n```bash\npython3.12 -m venv .venv\n.venv/bin/python -m pip install -r requirements-v3.lock.txt\n.venv/bin/python -m pip install --no-deps -e .\nPYTHONPATH=src MUJOCO_GL=egl .venv/bin/python -B scripts/render_frozen_v3.py --run {relative} --output-directory replay\n```\n\nAuthoritative portable XML/state/assets: `{relative}/delivery/static/portable/`; independent dynamic diagnostic model: `{relative}/physics/portable/`. Source/staging XML and provenance may retain historical absolute paths; these are not the portable loading entry. Search image cache is omitted, but all candidate logs, numerical metrics, API requests/responses, source snapshots and current source are included. The conserved task budget database is included; no new task or counter reset is implied by extraction. Fresh geometry fitting still requires its declared imported observations/Mount inputs.\n'''
    with tempfile.TemporaryDirectory(prefix='movens_v3_package_') as temp:
        snapshot=Path(temp)/'task.sqlite'
        with sqlite3.connect(db) as src,sqlite3.connect(snapshot) as dst:src.backup(dst)
        manifest[str(db.relative_to(ROOT))]=dict(sha256=sha(snapshot),bytes=snapshot.stat().st_size)
        with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as archive:
            for path in sorted(set(files)):archive.write(path,'Movens-Sim/'+str(path.relative_to(ROOT)))
            archive.write(snapshot,'Movens-Sim/'+str(db.relative_to(ROOT)));archive.writestr('Movens-Sim/DELIVERY_README.md',readme)
            archive.writestr('Movens-Sim/ZIP_CONTENTS.json',json.dumps(dict(created_utc=now(),files=manifest),indent=2))
        with zipfile.ZipFile(out) as archive:
            if archive.testzip():raise ValueError('Archive CRC verification failed')
    write(out.with_suffix('.manifest.json'),dict(archive=str(out),sha256=sha(out),bytes=out.stat().st_size,files=len(manifest),cache_excluded=True,credential_files_included=False))
    print('Packaged',len(manifest),'files,',round(out.stat().st_size/1024**2,2),'MiB:',out)


if __name__=='__main__':main()
