#!/usr/bin/env python3
"""Package code and three portable final scenes, keeping failures and verification evidence."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tarfile
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from movens_sim.util import ROOT,read,write,sha,digest,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();c=read(ROOT/a.config);root=ROOT/c['run_root'];out=root/'packages';out.mkdir(exist_ok=True);inventory=[]
    for ep in [0,1,2]:
        run=root/f'episode_{ep:06}';binding=read(run/'delivery_binding.json')
        if not (run/'reacceptance/final/acceptance.json').exists():raise ValueError('Independent reacceptance missing: '+str(ep))
        for name,h in binding['artifacts'].items():
            if sha(run/name)!=h:raise ValueError('Frozen artifact changed: '+name)
        target=out/f'episode_{ep:06}.tar.gz';temporary=target.with_suffix('.partial')
        with tarfile.open(temporary,'w:gz',compresslevel=3) as archive:
            for folder in ('delivery/static/portable','delivery/dynamic/portable','final_acceptance','reacceptance','masks','terra','mcp_plans','evidence','code_versions','revisions'):
                path=run/folder
                if path.exists():archive.add(path,arcname=f'episode_{ep:06}/'+folder)
            for path in run.iterdir():
                if path.is_file() and not path.name.startswith('.'):
                    archive.add(path,arcname=f'episode_{ep:06}/'+path.name)
            for path in (root/'logs').glob(f'episode_{ep:06}_*.log'):archive.add(path,arcname=f'episode_{ep:06}/logs/'+path.name)
        temporary.replace(target);inventory.append(dict(episode=ep,path=str(target.relative_to(root)),bytes=target.stat().st_size,sha256=sha(target),frozen_scene_hash=binding['scene_parameter_asset_hash']))
    changed=subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).splitlines();added=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines();names=sorted(set(changed+added));patch=root/'verification/implementation.patch';parts=[subprocess.check_output(['git','diff','--no-ext-diff'],cwd=ROOT)]
    for name in added:
        result=subprocess.run(['git','diff','--no-index','--','/dev/null',name],cwd=ROOT,capture_output=True);parts.append(result.stdout)
    patch.write_bytes(b'\n'.join(parts));target=out/'modified_code.tar.gz'
    with tarfile.open(target,'w:gz',compresslevel=3) as archive:
        for name in names:
            path=ROOT/name
            if path.is_file():archive.add(path,arcname=name)
        archive.add(patch,arcname='implementation.patch')
    provenance=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),changed_files={name:sha(ROOT/name) for name in names if (ROOT/name).is_file()},code_package_sha256=sha(target),code_package=str(target.relative_to(root)),episode_packages=inventory,created_utc=now(),policy_inference=False,training=False)
    provenance['working_tree_content_sha256']=digest(provenance['changed_files']);write(root/'code_delivery.json',provenance);write(out/'inventory.json',provenance)
    print(str(out))

if __name__=='__main__':main()
