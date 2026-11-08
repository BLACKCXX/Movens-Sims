#!/usr/bin/env python3
"""Publish a versioned repair snapshot for the authorized ep012 task only."""
import argparse
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from movens_sim.util import ROOT,CODE_ROOT,read,write,sha,digest,now
from movens_sim.reconstruction.workflow import identity

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--reason',required=True);a=p.parse_args();c=read(ROOT/a.config)
    if c['workflow']!='classified_reconstruction_v4' or c['allowed_episodes']!=[0,1,2]:raise ValueError('This publisher is restricted to the isolated ep012 workflow')
    root=ROOT/c['run_root'];sample=root/'episode_000000';ident=identity(sample,c);key=digest(ident['code'])[:16];target=root/'code_versions'/key
    if not target.exists():
        target.mkdir(parents=True)
        for folder in ('src','prompts','configs'):shutil.copytree(CODE_ROOT/folder,target/folder,ignore=shutil.ignore_patterns('__pycache__'))
        (target/'scripts').mkdir()
        for name in ('movens-sim','accept_reconstruction.py'):shutil.copy2(CODE_ROOT/'scripts'/name,target/'scripts'/name)
        shutil.copy2(CODE_ROOT/'pyproject.toml',target/'pyproject.toml')
    hashes={str(q.relative_to(target)):sha(q) for q in target.rglob('*') if q.is_file()};manifest=root/'code_versions'/(key+'_hashes.json');write(manifest,hashes)
    previous=[]
    for path in root.glob('episode_*/run_identity.json'):previous.append(digest(read(path)['code']))
    transition=dict(from_code_sha256s=sorted(set(previous)),to_code_sha256=digest(ident['code']),reason=a.reason,utc=now(),config_inputs_budgets_unchanged=True)
    write(root/'code_versions'/(key+'_transition.json'),transition);write(root/'rolling_code_transition.json',transition);write(root/'active_code.json',dict(code_root=str(target.relative_to(root)),hash_manifest=str(manifest.relative_to(root))))
    print(str(target))

if __name__=='__main__':main()
